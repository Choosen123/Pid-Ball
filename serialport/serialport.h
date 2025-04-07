#ifndef SERIALPORT_H
#define SERIALPORT_H

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/asio/serial_port.hpp>
#include <boost/endian/conversion.hpp>
#include <boost/endian/arithmetic.hpp>


using namespace boost::asio;
using namespace std;
using namespace boost::placeholders;

class MySerial{
private:
    string port_name;
    unsigned int baud_rate;
    unsigned int character_size;
    serial_port sp;
    int count=0;
    char data_buffer[4];
    
public:
    float angle;
    uint8_t position=75;
    
    MySerial(string name,unsigned int baud_rate,unsigned int character_size,io_context &io);

    bool init_and_open();

    void flush_buffer();

    void my_async_read();

    void my_async_write();

    void write_callback(const boost::system::error_code &ec,size_t bytes);

    void read_callback(const boost::system::error_code &ec,size_t bytes);

    void close_port();

    
};

#endif